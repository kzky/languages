import asyncio


async def count():
    print("One")
    await asyncio.sleep(1)
    print("Two")


async def count(i: int):
    print(i)
    await asyncio.sleep(1)
    j = i + 10
    print(j)
    return j

async def func():
    # on-completion is asyncio.as_completed()
    j0, j1, j2 = await asyncio.gather(count(0), count(1), count(2))
    return j0, j1, j2

if __name__ == '__main__':
    import time
    s = time.perf_counter()
    j0, j1, j2 = asyncio.run(func())
    print(j0, j1, j2)
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")

    
