from flask import Flask, Blueprint
from flask_restful import Api, Resource

app = Flask(__name__)
api_bp = Blueprint('api', "flask-restful with blueprint", url_prefix="/api",)
api = Api(api_bp)

class TodoItem(Resource):
    def get(self, id):
        return {'task': 'Say "Hello, World!" for {}'.format(id)}

api.add_resource(TodoItem, '/todos/<int:id>')
app.register_blueprint(api_bp)

if __name__ == '__main__':
    app.run(debug=True)

# call like
"""
curl -X GET "http://localhost:5000/api/todos/10"
"""
