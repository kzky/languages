#!/usr/bin/env python

from flask import Flask, request, jsonify
from flask.ext.login import LoginManager, UserMixin, login_required
from flask.ext.mongoengine import MongoEngine

# Init application
app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_key"
'''
def login_required(func):
    ....
    @wraps(func)
    def decorated_view(*args, **kwargs):
        if current_app.login_manager._login_disabled:
            return func(*args, **kwargs)
        elif not current_user.is_authenticated(): # Here, Runtime error occur
            return current_app.login_manager.unauthorized()
        return func(*args, **kwargs)
    return decorated_view
    ....

Runtime error occur
-----
current_user.is_authenticated()
RuntimeError: the session is unavailable because no secret key was set.  Set the secret_key on the application to something unique and secret.
-----
'''


# Init db
db = MongoEngine()
db_name = "flask-login-sample"
app.config['MONGODB_SETTINGS'] = {
    'db': db_name,
    # 'username':'webapp',
    # 'password':'pwd123'
}
db.init_app(app)
db.connection.drop_database(db_name)

# Init login manager
lm = LoginManager()
lm.init_app(app)

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

#@lm.unauthorized_handler
#def unauthorized():
#    # do stuff
#    data = {"result": "You need login"}
#    return jsonify(data)

# Controllers
@app.route("/hello")
def hello():
    return "Hello World!"

@app.route("/hello_protected")
@login_required
def hello_protected():
    """Can be called as the following.
    $ curl "http://localhost:5000/hello_protected?api_key=..."
    """
    print "hello_protected called"
    return "Hello World!"

@app.route("/add", methods=["POST"])
def add():
    """Called as the following.
    $ curl -H "Content-Type: application/json" -X POST -d '{"a":"10","b":"20"}' \
    http://localhost:5000/add
    """

    data = request.json
    ret = int(data["a"]) + int(data["b"])
    res = {"result": ret}
    return jsonify(res)

if __name__ == "__main__":
    # Create protected user
    api_key = "03094200e64fba27239b554d879d9ca654"
    user = User(name="me", api_key=api_key)
    user.save()
    app.run()
