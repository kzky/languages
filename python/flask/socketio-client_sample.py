#!/usr/bin/env python

from socketIO_client import SocketIO, BaseNamespace

class MyNamespace(BaseNamespace):

    def on_aaa_response(self, *args):
        print('on_aaa_response', args)

socketIO = SocketIO('localhost', 5000)
namespace = "my_namespace"
event_name = "my_event"
my_namespace = socketIO.define(MyNamespace, "/my_namespace")

my_namespace.emit(event_name, {"message": "from python socketio-client"})
socketIO.wait(seconds=1)
