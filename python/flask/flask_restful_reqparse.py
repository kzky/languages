from flask import Flask
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

# Hello World
class HelloWorld(Resource):

    parser = reqparse.RequestParser()
    parser.add_argument('rate', type=int, help='Rate cannot be converted')
    parser.add_argument('name', required=True)
    parser.add_argument('name_relocated', dest="name_dest")
    parser.add_argument('todos', type=str, action="append")
        
    def post(self):
        args = self.parser.parse_args()

        print args
        return args

api.add_resource(HelloWorld, '/hello')

if __name__ == '__main__':
    app.run(debug=True)
