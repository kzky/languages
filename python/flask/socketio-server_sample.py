#!/usr/bin/env python

from flask import Flask, render_template
from flask.ext.socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

event_name = "my_event"
namespace = "/my_namespace"

@socketio.on(event_name, namespace=namespace)
def test_message(message):
    print("called!")
    print("receive: ")
    print(message)
    #emit('my response', {'data': message['data']})

if __name__ == '__main__':
    socketio.run(app)

    
