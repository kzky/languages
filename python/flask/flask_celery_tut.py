from flask import Flask
from celery import Celery

def make_celery(app):
    celery = Celery(app.import_name,
                    broker=app.config['CELERY_BROKER_URL'],
                    backend=app.config["CELERY_RESULT_BACKEND"])
    celery.conf.update(app.config)
    TaskBase = celery.Task
    
    class ContextTask(TaskBase):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    return celery
    
flask_app = Flask(__name__)
flask_app.config.update(
    CELERY_BROKER_URL='amqp://localhost',
    CELERY_RESULT_BACKEND='amqp://localhost'
)
celery = make_celery(flask_app)

@celery.task
def add_together(a, b):
    return a + b
