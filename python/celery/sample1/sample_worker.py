from celery import Celery
import time
import celeryconfig

#app = Celery("tasks", backend="rpc", broker="amqp://guest@192.168.10.5")
app = Celery("tasks")
app.config_from_object(celeryconfig)

sleep_time = 10

@app.task
def add(x, y):
    time.sleep(sleep_time)
    return x + y
