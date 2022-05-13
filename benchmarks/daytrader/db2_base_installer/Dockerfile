FROM ubuntu:18.04

USER root

# Install prerequisite packages
RUN dpkg --add-architecture i386 \
	&& apt-get update && apt-get install -y --no-install-recommends \
		binutils file libaio1 liblogger-syslog-perl libnuma1 \
		libpam0g:i386 libstdc++6:i386 libxml2 zlib1g-dev \
	&& rm -rf "/var/lib/apt/lists/"*

# Copy and extract DB2 installer and response file
ADD "db2.tar.gz" "/"
COPY "db2server.rsp" "/"

# Install DB2 and cleanup unnecessary files
RUN "/server_dec/db2setup" -r "/db2server.rsp" \
	&& rm -rf "/server_dec" "/db2server.rsp" "/tmp/"* "/tmp/.com_ibm_tools_attach"

EXPOSE 50000

# Allow non-privileged user to modify db2 port in /etc/services
RUN chmod a+w "/etc" "/etc/services"

USER db2inst1

# Copy entrypoint script
COPY "run.sh" "/"

# Arguments: port [db_to_activate]
ENTRYPOINT ["/run.sh"]
