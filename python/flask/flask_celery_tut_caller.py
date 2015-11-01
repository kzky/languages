import flask_celery_tut

def main():
    result = flask_celery_tut.add_together.delay(10, 10)
    print result.wait()
    pass

if __name__ == '__main__':
    main()
