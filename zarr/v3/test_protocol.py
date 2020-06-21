import pytest

from zarr.storage import init_group
from zarr.v3 import MemoryStoreV3, RedisV3Store, V2from3Adapter, ZarrProtocolV3


async def test_scenario():

    store = MemoryStoreV3.sync()

    await store.async_set("data/a", bytes(1))

    with pytest.raises(ValueError):
        store.get("arbitrary")
    with pytest.raises(ValueError):
        store.get("data")
    with pytest.raises(ValueError):
        store.get("meta")  # test commit

    assert store.get("data/a") == bytes(1)
    assert await store.async_get("data/a") == bytes(1)

    await store.async_set("meta/this/is/nested", bytes(1))
    await store.async_set("meta/this/is/a/group", bytes(1))
    await store.async_set("meta/this/also/a/group", bytes(1))
    await store.async_set("meta/thisisweird/also/a/group", bytes(1))

    assert len(store.list()) == 5

    assert store.list_dir("meta/this") == ["meta/this", "meta/thisisweird"]
    assert await store.async_list_dir("meta/this") == ["meta/this", "meta/thisisweird"]


async def test_2():
    protocol = ZarrProtocolV3(MemoryStoreV3)
    store = protocol._store

    await protocol.async_create_group("g1")
    assert isinstance(await store.async_get("meta/g1.group"), bytes)


@pytest.mark.parametrize("klass", [MemoryStoreV3, RedisV3Store])
def test_misc(klass):

    _store = klass.sync()
    _store.initialize()
    store = V2from3Adapter(_store)

    init_group(store)

    if isinstance(_store, MemoryStoreV3):
        assert store._v3store._backend == {
            "meta/root.group": b'{\n    "zarr_format": "https://purl.org/zarr/spec/protocol/core/3.0"\n}'
        }
    assert store[".zgroup"] == b'{\n    "zarr_format": 2\n}'
