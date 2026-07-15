from hypothesis import given, strategies as st

from core import Machine
import numpy as np
from strategies import machine_strategies, usage_for_machine


@given(
    machine=machine_strategies(),
    row=st.data(),
)
def test_machine_allocation_with_usage_bigger_than_capacity(
    machine: Machine, row: st.DataObject
) -> None:
    r = row.draw(st.integers(0, machine.capacity.shape[0] - 1))
    c = row.draw(st.integers(0, machine.capacity.shape[1] - 1))
    usage = np.zeros_like(machine.capacity)
    usage[r, c] = machine.capacity[r, c] + 1
    assert not machine.add_usage(usage)


@given(
    machine=machine_strategies(),
    row=st.data(),
)
def test_machine_allocation_with_usage_smaller_or_equal_to_capacity_and_forward_time(
    machine: Machine, row: st.DataObject
) -> None:
    usage = row.draw(usage_for_machine(machine))
    assert np.all(usage <= machine.capacity)
    time_count = row.draw(st.integers(0, machine.capacity.shape[0] - 1))
    assert machine.add_usage(usage)

    expected = usage.copy()

    for time in range(time_count):
        machine.forward_time(time)

        expected[:, :-1] = expected[:, 1:]
        expected[:, -1] = 0

        np.testing.assert_array_equal(
            np.asarray(machine.usage),
            expected,
        )


# def test_machine_initialization()-> None:
# # TODO: create capacity and usage
# # add usage to machine, x times
# # and then foward time untill
# capacity = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int32)
# print(capacity.shape)
# print(capacity.ndim)
# usage =  np.array([[1, 1, 1], [1, 1, 1]], dtype=np.int32)
# machine = Machine(capacity)
# print(np.asarray(machine.capacity))
#
# machine.add_usage(usage)
#
# print(np.asarray(machine.usage))
# machine.forward_time(0)
# print(np.asarray(machine.usage))
# machine.forward_time(0)
# machine.forward_time(0)
# print(np.asarray(machine.usage))
#
#
