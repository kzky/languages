import cherrypy


@cherrypy.popargs('name')
class Band(object):
    def __init__(self):
        self.album = Album()

    @cherrypy.expose
    def index(self, name):
        return 'About %s...' % name

    @cherrypy.expose
    def albums(self, name):
        """
        """
        albums = ["a", "b", "c"]
        return 'Albums about %s are %s' % (name, str(albums))
        
        pass

    pass


@cherrypy.popargs('title')
class Album(object):
    def __init__(self):
        self.track = Track()
        pass

    @cherrypy.expose
    def index(self, name, title):
        return 'About %s by %s...' % (title, name)
    pass

    @cherrypy.expose
    def traks(self, name, title):
        tracks = ["a", "b", "c"]
        return 'Traks about %s by %s are %s' % (title, name, str(tracks))
    pass


@cherrypy.popargs('num', 'track')
class Track(object):

    @cherrypy.expose
    def index(self, name, title, num, track):
        return 'About %s by %s at num:%s,track:%s' % (title, name, num, track)

    @cherrypy.expose
    def detail(self, name, title, num, track, **params):
        """
        e.g.,
        $ curl -X GET
        "http://localhost:8080/nirvana/album/nevermind/track/06/polly/detail/"
        """
        arg1 = params["arg1"]
        arg2 = params["arg2"]
        print arg1, arg2
        return 'Detail About %s by %s at num:%s,track:%s' % (title, name, num, track)
        
        pass
    
if __name__ == '__main__':
    cherrypy.quickstart(Band())
