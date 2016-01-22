from sample_worker import add

# just call
add.delay(4, 4)

# ready
async_result = add.delay(4, 4)
async_result_bool = async_result.ready()
print "task is ready?: {}".format(async_result_bool)

# get
print async_result.get(timeout=10)
