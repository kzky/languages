from flask_restful import fields, marshal_with
from flask import Flask
from flask_restful import Resource, Api
from flask_restful import fields, marshal_with

# Restful Basics
app = Flask(__name__)
api = Api(app)

# fields module and marshal_with decorator
user_fields = {
    'id': fields.Integer,
    'name': fields.String,
}

resource_fields = {
    'task_id': fields.Integer,
    'task': fields.String(default="Default task"),
    'tasks': fields.List(fields.String(default="Default task")),
    'users': fields.List(fields.Nested(user_fields)),
    #'uri': fields.Url('todo_ep')
}

class TodoDao(object):
    def __init__(self, task_id, task, tasks, users):
        self.task_id = task_id
        self.task = task
        self.tasks = tasks
        self.users = users

        # This field will not be sent in the response (meaning can have other state)
        self.status = 'active'

class Todo(Resource):
    @marshal_with(resource_fields)
    def get(self, task_id):

        task_id = task_id
        task = 'Remember the milk'
        tasks = ["Remember the milk 000",
                 "Remember the milk 001",
                 "Remember the milk 002'"]

        users = [
            {"id": 0, "name": "kzky000"},
            {"id": 1, "name": "kzky001"},
            {"id": 2, "name": "kzky002"}
        ]
        return TodoDao(task_id=task_id, task=task, tasks=tasks, users=users)

api.add_resource(Todo, '/todos/<int:task_id>')

if __name__ == '__main__':
    app.run(debug=True)

# call like
"""
curl -X GET "http://localhost:5000/todos/10"
"""
