FROM db2-no-volumes

USER root

# Edit the entrypoint script to skip remounting the "/database"
# directory which is no longer a volume. Modify these lines to
# comment out the mount command and update the output message:
#
# echo "(*) Remounting ${STORAGE_DIR?} with suid... "
# mount -o remount,suid ${STORAGE_DIR?}
#
RUN sed -i -e 's/Remounting/NOT remounting/g' -e 's/^mount /# &/' \
	"/var/db2_setup/lib/setup_db2_instance.sh"

USER db2inst1
