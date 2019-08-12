

import sys
import logging
logging.basicConfig(stream=sys.stderr)

sys.path.insert(0, '/usr/local/apache2/htdocs/')

from diode_rest import app as application
