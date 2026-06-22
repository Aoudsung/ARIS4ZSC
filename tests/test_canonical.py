from tcaso_critic.canonical import canonical_hash, canonical_json


def test_canonical_hash_order_invariant():
    a = {"b": 2, "a": 1}
    b = {"a": 1, "b": 2}
    assert canonical_json(a) == canonical_json(b)
    assert canonical_hash(a) == canonical_hash(b)
