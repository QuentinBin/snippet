import logging
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
print(logger.getEffectiveLevel())
a = 10
logging.debug('matrix', np.eye(4))

logger.debug('logger debug 信息 ' % a)
logger.info('logger info 信息')
logger.warning('logger warn 信息')
logger.error('logger error 信息')
logger.critical('logger critical 信息')
logger.debug('logger %s 是自定义信息' % '这些东西')
