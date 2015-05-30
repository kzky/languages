#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging


def main():
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                        level=logging.DEBUG)
    logging.debug('degub')
    logging.info('info')
    logging.warning('warning')

if __name__ == '__main__':
    main()
