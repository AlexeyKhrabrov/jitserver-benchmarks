FROM db2-setup

USER root

# Allow non-privileged user to modify db2 port in /etc/services
RUN chmod a+w "/etc" "/etc/services"

USER db2inst1

# Copy entrypoint script
COPY "run.sh" "/"

# Arguments: port [db_to_activate]
ENTRYPOINT ["/run.sh"]
