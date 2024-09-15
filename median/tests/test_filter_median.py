from median.buffer import Buffer
from median.filter_median import FilterMedian

fm7 = FilterMedian(L=7, dtype=int, initB=[5, 7, 8, 8, 9, 10, 11])


def update_desc(buffer: Buffer, value: float | int):
    print(f"Insert: {value:>3}")
    print(f"Before: | {' | '.join([f'{i:>2}' for i in buffer.getBuffer()])} |")
    buffer.update(value)
    print(f"After : | {' | '.join([f'{i:>2}' for i in buffer.getBuffer()])} |")


def updateR_desc(buffer: Buffer, value: float | int):
    print(f"Insert: {value:>3}")
    print(buffer)
    ret = buffer.updateR(value)
    print(f"Dropped: {ret:>3}")
    print(buffer)


updateR_desc(fm7, 4)
updateR_desc(fm7, 6)
updateR_desc(fm7, 12)
updateR_desc(fm7, 6)
updateR_desc(fm7, 9)
updateR_desc(fm7, 11)
updateR_desc(fm7, 3)
