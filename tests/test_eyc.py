"""Tests for the curve, the pricer and the analytics.

Two kinds of assertion appear here. Some pin the implementation against values
the ECB published on a named date, which is an external reference. Others pin it
against identities that must hold whatever the data says. Both matter: the first
catches a wrong convention, the second catches a wrong formula that happens to
agree with the reference on a single day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from eyc import analysis, bonds, calibration, loader, strategies
from eyc.svensson import (
    annuity,
    discount_factor,
    forward_rate,
    instantaneous_forward,
    par_yield,
    spot_rate,
)

# ECB all-issuers curve, 21 August 2026, from the published YC dataset.
PARAMS_2026 = (
    1.9388025818,
    0.3033305102,
    1.3519631001,
    7.9195137498,
    0.6998681232,
    16.0641230843,
)
PUBLISHED_2026 = {"SR_10Y": 3.700447696, "IF_10Y": 4.584212225, "PY_10Y": 3.6371789603}

TOLERANCE_BP = 1e-4


def bp(a: float, b: float) -> float:
    return abs(a - b) * 100.0


class TestPublishedReference:
    """Agreement with the rates the ECB published, in the convention it uses."""

    def test_spot_matches_published(self):
        assert bp(spot_rate(10.0, *PARAMS_2026), PUBLISHED_2026["SR_10Y"]) < TOLERANCE_BP

    def test_forward_matches_published(self):
        value = instantaneous_forward(10.0, *PARAMS_2026)
        assert bp(value, PUBLISHED_2026["IF_10Y"]) < TOLERANCE_BP

    def test_par_yield_matches_published(self):
        value = par_yield(10.0, *PARAMS_2026, method="integral")
        assert bp(value, PUBLISHED_2026["PY_10Y"]) < TOLERANCE_BP

    def test_annual_compounding_would_be_wrong(self):
        """The exp(y) - 1 conversion is the classic error; pin its size."""
        continuous = spot_rate(10.0, *PARAMS_2026)
        annualised = (np.exp(continuous / 100.0) - 1.0) * 100.0
        assert bp(annualised, PUBLISHED_2026["SR_10Y"]) > 5.0

    def test_legacy_and_current_par_yield_differ(self):
        """The 2023 methodology change is real, not a rounding artefact."""
        current = par_yield(10.0, *PARAMS_2026, method="integral")
        legacy = par_yield(10.0, *PARAMS_2026, method="monthly")
        assert bp(current, legacy) > 0.4

    def test_legacy_par_yield_rejects_off_grid_maturity(self):
        with pytest.raises(ValueError):
            par_yield(10.01, *PARAMS_2026, method="monthly")


class TestModelIdentities:
    """Relations that hold by construction, independent of any published value."""

    @pytest.mark.parametrize("tau", [0.25, 1.0, 5.0, 10.0, 30.0])
    def test_spot_is_average_forward(self, tau):
        average = (
            quad(lambda s: instantaneous_forward(s, *PARAMS_2026), 0.0, tau, limit=200)[0]
            / tau
        )
        assert bp(average, spot_rate(tau, *PARAMS_2026)) < 1e-8

    def test_discount_factor_at_zero_is_one(self):
        assert abs(discount_factor(0.0, *PARAMS_2026) - 1.0) < 1e-12

    def test_discount_factor_is_positive(self):
        taus = np.linspace(0.01, 30.0, 200)
        assert np.all(discount_factor(taus, *PARAMS_2026) > 0)

    def test_short_end_limit(self):
        """As maturity goes to zero, spot and forward both tend to beta0 + beta1."""
        limit = PARAMS_2026[0] + PARAMS_2026[1]
        assert bp(spot_rate(1e-9, *PARAMS_2026), limit) < 1e-6
        assert bp(instantaneous_forward(1e-9, *PARAMS_2026), limit) < 1e-6

    def test_annuity_matches_quadrature(self):
        for tau in (1.0, 10.0, 30.0):
            reference = quad(
                lambda s: discount_factor(s, *PARAMS_2026), 0.0, tau, limit=200
            )[0]
            assert abs(annuity(tau, *PARAMS_2026) - reference) < 1e-12

    def test_forward_rate_composes(self):
        """Compounding 0->5 then 5->10 must reproduce the 0->10 spot."""
        near = spot_rate(5.0, *PARAMS_2026) * 5.0
        far = forward_rate(5.0, 10.0, *PARAMS_2026) * 5.0
        assert bp((near + far) / 10.0, spot_rate(10.0, *PARAMS_2026)) < 1e-9

    def test_vectorises(self):
        taus = np.array([1.0, 10.0, 30.0])
        vector = spot_rate(taus, *PARAMS_2026)
        scalars = [spot_rate(t, *PARAMS_2026) for t in taus]
        assert np.allclose(vector, scalars, atol=1e-15)


class TestBondPricing:
    def test_par_coupon_prices_at_par(self):
        """The bond pricer and the curve must agree, or one of them is wrong."""
        for maturity in (2.0, 5.0, 10.0, 30.0):
            times = np.arange(1, int(maturity) + 1, dtype=float)
            discounts = discount_factor(times, *PARAMS_2026)
            coupon = 100.0 * (1.0 - discounts[-1]) / discounts.sum()
            bond = bonds.Bond(maturity, coupon)
            assert abs(bonds.dirty_price(bond, PARAMS_2026) - 100.0) < 1e-10

    def test_yield_round_trip(self):
        bond = bonds.Bond(10.0, 3.0)
        dirty = bonds.dirty_price(bond, PARAMS_2026)
        ytm = bonds.yield_to_maturity(bond, dirty)
        assert abs(bonds.price_from_yield(bond, ytm) - dirty) < 1e-9

    def test_dv01_consistent_with_duration(self):
        """DV01 must equal effective duration times price times one basis point."""
        bond = bonds.Bond(10.0, 3.0)
        price = bonds.dirty_price(bond, PARAMS_2026)
        implied = bonds.effective_duration(bond, PARAMS_2026) * price * 1e-4
        assert abs(bonds.dv01(bond, PARAMS_2026) - implied) < 1e-9

    def test_zero_coupon_duration_equals_maturity(self):
        """Exact in the limit; the residual is the central difference truncation.

        A one basis point bump leaves a second order term of order bump squared
        over six, about 1.7e-6 years here, which is why the tolerance is 1e-5 and
        not machine precision.
        """
        bond = bonds.Bond(10.0, 0.0)
        assert abs(bonds.effective_duration(bond, PARAMS_2026) - 10.0) < 1e-5

    def test_convexity_is_positive(self):
        assert bonds.effective_convexity(bonds.Bond(30.0, 2.0), PARAMS_2026) > 0

    def test_accrued_between_coupons(self):
        """A bond 0.3 years past its coupon has accrued 0.3 of a year of interest."""
        bond = bonds.Bond(9.7, 3.0)
        assert abs(bond.accrued() - 3.0 * 0.3) < 1e-9
        assert bonds.clean_price(bond, PARAMS_2026) < bonds.dirty_price(
            bond, PARAMS_2026
        )

    def test_longer_bond_has_more_duration(self):
        short = bonds.effective_duration(bonds.Bond(2.0, 3.0), PARAMS_2026)
        long = bonds.effective_duration(bonds.Bond(30.0, 3.0), PARAMS_2026)
        assert long > short

    def test_duration_families_disagree_on_a_sloped_curve(self):
        """Modified and effective duration are different measures, not synonyms."""
        report = bonds.risk_report(bonds.Bond(10.0, 3.0), PARAMS_2026)
        assert abs(report["duration_gap"]) > 0.1

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            bonds.Bond(-1.0, 3.0)
        with pytest.raises(ValueError):
            bonds.Bond(10.0, 3.0, frequency=3)


class TestCarryRolldown:
    def test_decomposition_adds_up(self):
        result = analysis.carry_rolldown(PARAMS_2026, 10.0, 0.25)
        total = result["carry_bp"] + result["rolldown_bp"]
        assert abs(total - result["total_return_bp"]) < 1e-8

    def test_rolldown_positive_on_upward_curve(self):
        """The 2026 curve slopes up, so a ten year position rolls down into profit."""
        assert analysis.carry_rolldown(PARAMS_2026, 10.0, 0.25)["rolldown_bp"] > 0

    def test_rejects_horizon_beyond_maturity(self):
        with pytest.raises(ValueError):
            analysis.carry_rolldown(PARAMS_2026, 1.0, 2.0)


class TestStrategies:
    """The trading layer, tested on synthetic data so the assertions are exact."""

    @staticmethod
    def _loadings():
        taus = np.array([1.0, 2.0, 5.0, 10.0, 30.0])
        level = np.ones_like(taus)
        slope = np.log(taus) - np.log(taus).mean()
        curvature = -((np.log(taus) - np.log(taus).mean()) ** 2)
        matrix = np.column_stack([level, slope, curvature])
        return matrix / np.linalg.norm(matrix, axis=0)

    def test_neutral_weights_kill_level_and_slope(self):
        """The whole point of the structure: no level or slope exposure left."""
        loadings = self._loadings()
        weights = strategies.neutral_weights(loadings, [1, 2, 3])
        exposure = weights @ loadings[[1, 2, 3], :2]
        assert np.allclose(exposure, 0.0, atol=1e-12)

    def test_naive_butterfly_is_not_neutral(self):
        """2:-1:-1 leaves slope risk, which is why it is not used here."""
        loadings = self._loadings()
        naive = np.array([-1.0, 2.0, -1.0])
        assert abs(naive @ loadings[[1, 2, 3], 1]) > 1e-6

    def test_conditioning_flags_collinear_legs(self):
        """Legs close together in factor space give a worse condition number."""
        loadings = self._loadings()
        spread_out = strategies.conditioning(loadings, [0, 2, 4])
        bunched = strategies.conditioning(loadings, [2, 3, 4])
        assert bunched > spread_out

    def test_structure_change_is_not_the_level_difference(self):
        """Constant-weight P&L must ignore the jump caused by re-solving weights."""
        dates = pd.date_range("2020-01-01", periods=4, freq="B")
        rates = pd.DataFrame(
            {2.0: [1.0, 1.0, 1.0, 1.0], 5.0: [2.0, 2.0, 2.0, 2.0],
             10.0: [3.0, 3.0, 3.0, 3.0]},
            index=dates,
        )
        weights = pd.DataFrame(
            {"w1": [-0.5, -0.9, -0.9, -0.9], "wb": [1.0, 1.0, 1.0, 1.0],
             "w2": [-0.5, -0.1, -0.1, -0.1]},
            index=dates,
        )
        change = strategies.structure_change(rates, weights, (2.0, 5.0, 10.0))
        # Rates never move, so nothing was earned even though the weights did.
        assert np.allclose(change.to_numpy(), 0.0, atol=1e-12)

    def test_half_life_recovers_a_known_process(self):
        rng = np.random.default_rng(0)
        phi, size = 0.98, 20000
        series = np.zeros(size)
        for i in range(1, size):
            series[i] = phi * series[i - 1] + rng.normal()
        expected = -np.log(2) / np.log(phi)
        estimated = strategies.half_life(pd.Series(series))
        assert abs(estimated - expected) / expected < 0.25

    def test_backtest_earns_the_next_day_move(self):
        """No look-ahead: a position taken on day t earns the move t to t+1."""
        change = pd.Series([1.0, -2.0, 3.0], index=pd.date_range("2020-01-01", periods=3))
        position = pd.Series([1.0, 1.0, 0.0], index=change.index)
        frame = strategies.backtest(change, position, cost_bp=0.0)
        assert frame["gross_pnl_bp"].tolist() == [-1.0, 2.0, -0.0]

    def test_costs_reduce_pnl(self):
        change = pd.Series([1.0, -2.0, 3.0], index=pd.date_range("2020-01-01", periods=3))
        position = pd.Series([1.0, -1.0, 1.0], index=change.index)
        free = strategies.backtest(change, position, cost_bp=0.0)
        charged = strategies.backtest(change, position, cost_bp=1.0)
        assert charged["net_pnl_bp"].sum() < free["net_pnl_bp"].sum()

    def test_rejects_wrong_leg_count(self):
        with pytest.raises(ValueError):
            strategies.neutral_weights(self._loadings(), [0, 1])


class TestCarry:
    """Roll down of a structure, and the filter built on it."""

    FLAT = (3.0, 0.0, 0.0, 0.0, 1.0, 10.0)

    def _frames(self, periods: int):
        dates = pd.date_range("2020-01-01", periods=periods, freq="B")
        params = pd.DataFrame(
            dict(zip(loader.PARAMS, [[v] * periods for v in self.FLAT])), index=dates
        )
        weights = pd.DataFrame(
            {
                "w1": [-0.5] * periods,
                "wb": [1.0] * periods,
                "w2": [-0.5] * periods,
            },
            index=dates,
        )
        return params, weights

    def test_flat_curve_has_no_roll(self):
        """Nothing to roll down when every maturity carries the same rate."""
        params, weights = self._frames(3)
        roll = strategies.structure_rolldown(params, weights, (2.0, 5.0, 10.0), 0.25)
        assert np.allclose(roll.to_numpy(), 0.0, atol=1e-12)

    def test_rejects_horizon_beyond_front_leg(self):
        params, weights = self._frames(2)
        with pytest.raises(ValueError):
            strategies.structure_rolldown(params, weights, (2.0, 5.0, 10.0), 3.0)

    def test_reversion_fraction_at_one_half_life(self):
        assert abs(strategies.reversion_fraction(60.0, 60.0) - 0.5) < 1e-12
        # A series that never reverts closes none of its gap, so the share is
        # zero. Returning one here would let the carry filter through exactly
        # the structures with no edge to capture.
        assert strategies.reversion_fraction(float("inf"), 60.0) == 0.0

    def test_carry_filter_declines_negative_expectation(self):
        """A long whose roll outweighs the reversion it can capture is dropped."""
        index = pd.date_range("2020-01-01", periods=3)
        position = pd.Series([1.0, 1.0, -1.0], index=index)
        deviation = pd.Series([10.0, 10.0, -10.0], index=index)
        roll = pd.Series([1.0, 9.0, -9.0], index=index)
        kept = strategies.carry_filter(position, deviation, roll, reversion=0.5)
        assert kept.tolist() == [1.0, 0.0, 0.0]


class TestKeyRateDurations:
    def test_buckets_sum_to_effective_duration(self):
        """The tent basis sums to one, so the buckets must sum to the total."""
        for bond in (
            bonds.Bond(2.0, 2.5),
            bonds.Bond(10.0, 3.0),
            bonds.Bond(30.0, 2.0),
        ):
            buckets = bonds.key_rate_durations(bond, PARAMS_2026)
            total = bonds.effective_duration(bond, PARAMS_2026)
            assert abs(buckets.sum() - total) < 1e-5

    def test_zero_coupon_risk_sits_in_its_own_bucket(self):
        """A ten year zero has no cash flow anywhere else, so nor has its risk."""
        buckets = bonds.key_rate_durations(bonds.Bond(10.0, 0.0), PARAMS_2026)
        assert buckets[10.0] > 9.9
        assert buckets.drop(10.0).abs().max() < 1e-9

    def test_long_bond_is_not_hedged_by_the_ten_year(self):
        """The point of the exercise: a 10Y hedge misses most of a 30Y bond."""
        buckets = bonds.key_rate_durations(bonds.Bond(30.0, 2.0), PARAMS_2026)
        assert buckets[10.0] / buckets.sum() < 0.25


class TestCalibration:
    def test_par_yield_of_a_flat_curve(self):
        """On a flat continuous curve the coupon-equivalent par yield is exact.

        A flat zero rate y prices every coupon at exp(-y t), and the par coupon
        collapses to 2 * (exp(y/2) - 1), the semiannual equivalent of y.
        """
        rate = 3.0
        flat = (rate, 0.0, 0.0, 0.0, 1.0, 10.0)
        expected = 2.0 * (np.exp(rate / 100.0 / 2.0) - 1.0) * 100.0
        values = calibration.model_par_yields([1.0, 5.0, 10.0, 30.0], flat)
        assert np.allclose(values, expected, atol=1e-10)

    def test_rejects_off_grid_tenors(self):
        with pytest.raises(ValueError):
            calibration.model_par_yields([1.3], PARAMS_2026)

    def test_swapping_the_time_constants_changes_the_curve(self):
        """Svensson has no swap symmetry, and assuming one corrupts a solution.

        beta1 and beta2 share tau1, so exchanging (beta2, tau1) with (beta3, tau2)
        moves the beta1 term too. Reordering solutions on that false symmetry
        silently rewrites the curve they describe.
        """
        one = (2.0, -1.0, 3.0, -2.0, 1.5, 9.0)
        swapped = (2.0, -1.0, -2.0, 3.0, 9.0, 1.5)
        grid = np.array([0.5, 2.0, 10.0, 30.0])
        gap = np.abs(spot_rate(grid, *one) - spot_rate(grid, *swapped)).max()
        assert gap > 0.1

    def test_swap_symmetry_holds_only_when_beta1_is_zero(self):
        """The special case where the exchange really is a symmetry."""
        one = (2.0, 0.0, 3.0, -2.0, 1.5, 9.0)
        swapped = (2.0, 0.0, -2.0, 3.0, 9.0, 1.5)
        grid = np.array([0.5, 2.0, 10.0, 30.0])
        assert np.allclose(spot_rate(grid, *one), spot_rate(grid, *swapped), atol=1e-12)

    def test_recovers_a_well_identified_curve(self):
        """Time constants far apart: the fit finds the curve."""
        truth = (4.0, -2.0, 3.0, -1.5, 1.2, 12.0)
        outcome = calibration.recovery_test(np.arange(1, 31, dtype=float), truth)
        assert outcome["curve_error_bp"] < 0.01


class TestLoaderContracts:
    def test_maturity_codes(self):
        assert loader.maturity_code(0.25) == "3M"
        assert loader.maturity_code(1.0) == "1Y"
        assert loader.maturity_code(1.25) == "1Y3M"
        assert loader.maturity_code(30.0) == "30Y"

    def test_rate_codes_cover_published_grid(self):
        codes = loader.rate_codes("SR")
        assert len(codes) == len(loader.MATURITIES) == 120
        assert codes[0] == "SR_3M"
        assert codes[-1] == "SR_30Y"

    def test_rejects_unknown_inputs(self):
        with pytest.raises(ValueError):
            loader.rate_codes("XX")
        with pytest.raises(ValueError):
            loader.fetch(["BETA0"], curve="JUNK")
