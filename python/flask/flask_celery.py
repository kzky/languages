#!/usr/bin/env python

from celery import Celery
from flask import Flask, request, jsonify
from flask.ext.login import LoginManager, UserMixin, login_required
from flask.ext.mongoengine import MongoEngine
import time

def make_celery(app):
    celery = Celery(app.import_name, broker=app.config['CELERY_BROKER_URL'])
    celery.conf.update(app.config)
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    return celery

# Init application
app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_key"

# Init db
db = MongoEngine()
db_name = "flask-celery-sample00"
app.config['MONGODB_SETTINGS'] = {
    'db': db_name,
    # 'username':'webapp',
    # 'password':'pwd123'
}
db.init_app(app)
#db.connection.drop_database(db_name)

# Init login manager
lm = LoginManager()
lm.init_app(app)

# Init celery
app.config['CELERY_BROKER_URL'] = 'amqp://guest@localhost'
celery = make_celery(app)

# Models
class User(db.Document, UserMixin):
    name = db.StringField(unique=True, required=True)
    email = db.StringField(unique=True)
    api_key = db.StringField(unique=True)

    def get_id(self):
        return unicode(self.id)

# some related to LoginManager
@lm.user_loader
def load_user(userid):
    print "load_user called"
    user = User.objects(id=userid).first()
    if user is None:
        return None
    return user

@lm.request_loader
def load_user_from_request(request):
    # first, try to login using the api_key url arg
    api_key = request.args.get('api_key')
    if api_key is not None:
        print "api_key: {}".format(api_key)
        user = User.objects(api_key=api_key).first()
        if user:
            return user

    return None

# Celery tasks
@celery.task
def add_task(a, b):
    ret = a + b
    time.sleep(3)
    print ret

    return ret

@celery.task
def get_user_task(name):
    print app
    print db
    print db.connection
    print dir(db.connection)
    print User.objects(name=name)
    print User.objects(name=name).first()
    user = User.objects(name=name).first()
    
    print user.name
     
    return user.name

# Controller
@app.route("/add", methods=["POST"])
def add():
    data = request.json

    import flask_celery
    flask_celery.add_task.delay(int(data["a"]), int(data["b"]))
    #celery.add_task.delay(int(data["a"]), int(data["b"]))
    

    res = {"result": "received"}
    return jsonify(res)

# Controller
@app.route("/user/<name>", methods=["GET"])
def get_uesr(name):
    #user = User.objects(name=name).first()
    #print "username: {}".format(user.name)
    print name
    
    import flask_celery
    flask_celery.get_user_task.delay(name)
    
    res = {"result": "received"}
    return jsonify(res)

if __name__ == "__main__":

    # call Celery like
    # $ celery -A flask_celery.celery worker
    
    # Create protected user
    api_key = "03094200e64fba27239b554d879d9ca654"
    user = User(name="me", api_key=api_key)
    user_ = User.objects(name="me").first()
    if user_ is None:
        user.save()
    print dir()

    print add_task(10, 10)  # Can call a task here, so that we can unittest tasks without worker and broker.
    app.run()

