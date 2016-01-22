from sample_worker import add

# Call asynchronously
result = add.delay(4)

# Get
print result.get()

