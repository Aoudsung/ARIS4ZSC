"""Partner generator geometry tests (METHOD_SPEC §7.1/§7.2).

Specification entries covered:
- §7.1 the code space is the 3-dimensional tetrahedral simplex: four
  affinely independent unit anchors with pairwise inner products -1/3,
  barycentric maps stay convex, and neighbouring codes stay inside the
  fitted simplex support (the legacy cube clipping is abolished).
- §7.2 the retained code archive keeps the top-256 diversity contributions,
  and the shared-state diversity bank measures mean pairwise signature
  distance across codes on identical environment states.
"""

from __future__ import annotations

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.path_c.generator_training import (  # noqa: E402
    archive_insert,
    bank_pairwise_signature_diversity,
    barycentric_coordinates,
    codes_from_barycentric,
    empty_code_archive,
    fit_dirichlet_alpha,
    simplex_neighbor_codes,
    source_code_anchors,
)


def test_source_code_anchors_form_a_regular_tetrahedron() -> None:
    # §7.1: vertices are unit vectors with pairwise inner products -1/3,
    # i.e. affinely independent (rank 3) unlike the legacy rank-2 pair.
    anchors = np.asarray(source_code_anchors(4, 3))
    assert anchors.shape == (4, 3)
    np.testing.assert_allclose(np.linalg.norm(anchors, axis=-1), 1.0, atol=1e-6)
    gram = anchors @ anchors.T
    off_diagonal = gram[~np.eye(4, dtype=bool)]
    np.testing.assert_allclose(off_diagonal, -1.0 / 3.0, atol=1e-6)
    centered = anchors - anchors.mean(axis=0)
    assert np.linalg.matrix_rank(centered) == 3


def test_source_code_anchors_reject_legacy_shapes() -> None:
    with pytest.raises(ValueError, match="SP/OP/SA/FCP"):
        source_code_anchors(2, 3)
    with pytest.raises(ValueError, match="code_dim must be 3"):
        source_code_anchors(4, 8)


def test_barycentric_maps_are_convex_and_invertible() -> None:
    key = jax.random.PRNGKey(11)
    weights = jax.random.dirichlet(key, jnp.ones(4), shape=(5,))
    codes = codes_from_barycentric(weights)
    recovered = barycentric_coordinates(codes)
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(weights), atol=1e-4)
    round_trip = codes_from_barycentric(recovered)
    np.testing.assert_allclose(np.asarray(round_trip), np.asarray(codes), atol=1e-4)
    # Convexity: weights are non-negative and sum to one.
    assert float(np.min(np.asarray(recovered))) >= 0.0
    np.testing.assert_allclose(
        np.sum(np.asarray(recovered), axis=-1), np.ones(5), atol=1e-5
    )


def test_simplex_neighbor_codes_stay_inside_the_support() -> None:
    # §7.1: perturbations are convex combinations of the four anchors, so no
    # neighbour can leave the fitted simplex support.
    key = jax.random.PRNGKey(13)
    weights = jax.random.dirichlet(key, jnp.ones(4), shape=(8,))
    codes = codes_from_barycentric(weights)
    neighbours = simplex_neighbor_codes(codes, step=0.05)
    neighbour_weights = np.asarray(barycentric_coordinates(neighbours))
    assert float(neighbour_weights.min()) >= 0.0
    np.testing.assert_allclose(
        neighbour_weights.sum(axis=-1), np.ones(8), atol=1e-5
    )
    # Moving towards the centroid shrinks extreme weights.
    original = np.asarray(barycentric_coordinates(codes))
    assert float(neighbour_weights.max()) <= float(original.max()) + 1e-6


def test_fit_dirichlet_alpha_is_positive_and_concentration_ordered() -> None:
    # Codes concentrated near one vertex fit a Dirichlet with a dominant
    # concentration component; all components stay strictly positive.
    anchors = source_code_anchors(4, 3)
    near_vertex = jnp.stack(
        [codes_from_barycentric(jnp.asarray([0.9, 0.04, 0.03, 0.03]))] * 32
    )
    key = jax.random.PRNGKey(17)
    jitter = 0.01 * jax.random.normal(key, near_vertex.shape)
    alpha = np.asarray(fit_dirichlet_alpha(near_vertex + jitter, anchors))
    assert alpha.shape == (4,)
    assert float(alpha.min()) > 0.0
    assert int(np.argmax(alpha)) == 0


def test_code_archive_keeps_the_top_contributions() -> None:
    # §7.2: capacity 256; insertion keeps the largest diversity contributions.
    archive = empty_code_archive(capacity=4)
    assert int(np.asarray(archive.count)) == 0
    codes = jnp.asarray(np.eye(4, 3, dtype=np.float32))[:4]
    contributions = jnp.asarray([0.1, 0.5, 0.3, 0.2], dtype=jnp.float32)
    archive = archive_insert(archive, codes, contributions)
    assert int(np.asarray(archive.count)) == 4
    kept = np.asarray(archive.contributions)
    np.testing.assert_allclose(kept, np.sort(kept)[::-1], atol=1e-6)
    # Overflow: only the top-4 contributions survive.
    more_codes = jnp.asarray(np.eye(3, dtype=np.float32))
    more_contributions = jnp.asarray([0.9, 0.05, 0.7], dtype=jnp.float32)
    archive = archive_insert(archive, more_codes, more_contributions)
    assert archive.codes.shape == (4, 3)
    assert int(np.asarray(archive.count)) == 4
    kept = np.asarray(archive.contributions)
    np.testing.assert_allclose(kept, [0.9, 0.7, 0.5, 0.3], atol=1e-6)


def test_bank_diversity_is_zero_for_identical_codes_and_positive_elsewhere(
) -> None:
    # §7.2: diversity is the mean pairwise code distance over real signatures
    # measured on identical bank states; identical codes cannot earn diversity.
    identical = jnp.stack(
        [jnp.ones((4, 6)), jnp.ones((4, 6))], axis=0
    )
    assert float(np.asarray(bank_pairwise_signature_diversity(identical))) == (
        pytest.approx(0.0, abs=1e-5)
    )
    distinct = jnp.stack(
        [jnp.ones((4, 6)), -jnp.ones((4, 6))], axis=0
    )
    diversity = float(np.asarray(bank_pairwise_signature_diversity(distinct)))
    assert diversity > 0.0
    # Centered signatures: adding a shared offset does not change diversity.
    shifted = distinct + 7.0
    assert float(np.asarray(bank_pairwise_signature_diversity(shifted))) == (
        pytest.approx(diversity, abs=1e-5)
    )
