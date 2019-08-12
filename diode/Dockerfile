FROM httpd:2.4
COPY ./diode_2.0/client/ /usr/local/apache2/htdocs/client/
COPY ./diode/ /usr/local/apache2/htdocs/client/renderer_dir/
COPY ./diode_2.0/diode2.wsgi /usr/local/apache2/htdocs/diode2.wsgi
COPY ./diode_2.0/diode_rest.py /usr/local/apache2/htdocs/diode_rest.py

COPY . /usr/local/apache2/htdocs/installer/

RUN apt update && apt install apt-utils python3 python3-dev python3-pip gcc g++ -y --no-install-recommends
RUN pip3 install --upgrade pip
RUN pip3 install setuptools
RUN pip3 install mod_wsgi
RUN mod_wsgi-express install-module >> /usr/local/apache2/conf/httpd.conf

# Install the WSGI program
RUN echo "WSGIDaemonProcess diode2 user=pythonuser group=pythonuser home="/usr/local/apache2/htdocs/" threads=2" >> /usr/local/apache2/conf/httpd.conf
RUN echo "WSGIApplicationGroup %{GLOBAL}" >> /usr/local/apache2/conf/httpd.conf
RUN echo "WSGIScriptAlias /system /usr/local/apache2/htdocs/diode2.wsgi" >> /usr/local/apache2/conf/httpd.conf

# There is a difference to the development server: The base url changed (added /system) - adjust that in the config
RUN sed -i -e "s/window.location.host;/window.location.host + \"\/system\";/g" /usr/local/apache2/htdocs/client/main.js && cat /usr/local/apache2/htdocs/client/main.js

# There is more config in some cases - #TODO: Check if those are actually necessary

# Install dace (does not install DIODE2 (yet) )
#RUN pip3 install dace

# The pip DIODE is likely to be outdated - install from fresh (copied) data
RUN cd /usr/local/apache2/htdocs/installer/ && pip3 install --upgrade . && rm -r ../installer

# Install DaCe dependencies (might be too much)
RUN apt install libyaml-dev libgirepository1.0-dev xdot python3-gi -y

# Install PAPI
RUN apt install libpapi-dev -y

# Theoretically unnecessary - there is a include issue in DIODE1
#RUN apt install libgtksourceviewmm-3.0-dev python3-cairo python3-gi-cairo -y
#RUN pip3 install pygobject matplotlib

# Since the local paths are massively broken in this instance, the Working Directories / Config files are in out-of-permission regions. Try to add a new user with its own home for this
RUN useradd -m pythonuser

# User path is not enough - python tries to write to /root, fails due to permissions. Add those
RUN chmod 777 /root
RUN chmod -R 777 /home/pythonuser

# Stuff writes to the dist-packages. So unlock it all
RUN chmod -R 777 /usr/local/lib/python*/dist-packages/dace/

# Unlock the web directory (Allow the REST Server to create files in this directory)
RUN chmod -R 777 /usr/local/apache2/
RUN chmod -R 777 /tmp/
