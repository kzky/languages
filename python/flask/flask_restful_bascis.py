from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

# Hello World
class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world by get'}

    def post(self):
        return {'hello': 'world by post'}

    def put(self):
        return {'hello': 'world by put'}

    def delete(self):
        return {'hello': 'world by delete'}
api.add_resource(HelloWorld, '/hello')

# Call like
"""
curl -X GET "http://localhost:5000/hello"
curl -X POST "http://localhost:5000/hello"
curl -X put "http://localhost:5000/hello"
curl -X DELETE "http://localhost:5000/hello"
"""

# Todo Sample
todos = {}
class TodoSimple(Resource):
    def get(self, todo_id):
        return {todo_id: todos[todo_id]}

    def put(self, todo_id):
        todos[todo_id] = request.form['data']
        return {todo_id: todos[todo_id]}


api.add_resource(TodoSimple, '/<string:todo_id>')

if __name__ == '__main__':
    app.run(debug=True)

# Call like
"""
curl http://localhost:5000/todo1 -d "data=Remember the milk" -X PUT
curl http://localhost:5000/todo1
curl http://localhost:5000/todo2 -d "data=Change my brakepads" -X PUT
curl http://localhost:5000/todo2
"""



