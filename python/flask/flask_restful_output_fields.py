from flask_restful import fields, marshal_with
from flask import Flask
from flask_restful import Resource, Api
from flask_restful import fields, marshal_with

# Restful Basics
app = Flask(__name__)
api = Api(app)

# fields module and marshal_with decorator
resource_fields = {
    'task_id': fields.Integer,
    'task': fields.String(default="Default task"),
    #'uri': fields.Url('todo_ep')
}

class TodoDao(object):
    def __init__(self, task_id, task):
        self.task_id = task_id
        self.task = task

        # This field will not be sent in the response
        self.status = 'active'

class Todo(Resource):
    @marshal_with(resource_fields)
    def get(self, task_id):
        return TodoDao(task_id=task_id, task='Remember the milk')

api.add_resource(Todo, '/todos/<int:task_id>')

if __name__ == '__main__':
    app.run(debug=True)

# call like
"""
curl -X GET "http://localhost:5000/todos/10"
"""
