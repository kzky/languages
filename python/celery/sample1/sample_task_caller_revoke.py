from sample_worker import add
import time
import celery


# delay
async_result = add.delay(4, 4)
print async_result.id
time.sleep(2)

# revoke
async_result.revoke(terminate=True)  # otherwise, restart

time.sleep(2)

# delay2
async_result = add.delay(4, 4)
print async_result.id
time.sleep(2)

# revoke2
celery.task.control.revoke(async_result.id, terminate=True)  # otherwise, restart

time.sleep(2)
