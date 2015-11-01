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

