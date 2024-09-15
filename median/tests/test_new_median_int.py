from median.median import Median

nm7 = Median(L=7, dtype=int, initB=[5, 7, 8, 8, 9, 10, 11])


def update_desc(buffer: Median, value: float | int):
    print(f"Insert: {value:>3}")
    print(f"Before: | {' | '.join([f'{i:>2}' for i in buffer.getBuffer()])} |")
    buffer.update(value)
    print(f"After : | {' | '.join([f'{i:>2}' for i in buffer.getBuffer()])} |")


def updateR_desc(buffer: Median, value: float | int):
    print(
        f"Insert: {value:>3} | Buffer: | {' | '.join([f'{i:>2}' for i in buffer.getBuffer()])} |"
    )
    ret = buffer.updateR(value)
    print(
        f"Drop  : {ret:>3} | Buffer: | {' | '.join([f'{i:>2}' for i in buffer.getBuffer()])} |"
    )


def updateI_desc(buffer: Median, value: float | int):
    print(f"Insert: {value:>3}")
    print(f"Before: | {' | '.join([f'{i:>2}' for i in buffer.getBuffer()])} |")
    buffer.update_inef(value)
    print(f"After : | {' | '.join([f'{i:>2}' for i in buffer.getBuffer()])} |")


updateR_desc(nm7, 4)
updateR_desc(nm7, 6)
updateR_desc(nm7, 12)
updateR_desc(nm7, 6)
updateR_desc(nm7, 9)
updateR_desc(nm7, 11)
updateR_desc(nm7, 3)
