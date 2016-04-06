import sys

def main():
    if len(sys.argv) < 2:
        filepath = "./dataset/ratings_3cols.dat"
    else:
        filepath = sys.argv[1]

    print filepath
    
    pass

if __name__ == '__main__':
    main()
