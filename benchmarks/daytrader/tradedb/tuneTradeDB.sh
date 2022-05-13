#!/bin/bash

set -e -u -o pipefail

. ~/"sqllib/db2profile"


db="tradedb"

if (( $# >= 1 )); then
	log_path="${1}"
else
	log_path=~/"TradeDBLog"
fi


#db2set DB2_APM_PERFORMANCE=
#db2set DB2_NO_PKG_LOCK=

# New tunings
db2set DB2_SKIPINSERTED=on
db2set DB2_MAX_NON_TABLE_LOCKS=1000
db2set DB2_SKIP_LOG_WAIT=YES

# For tmpfs
#db2set DB2_DIRECT_IO=OFF
#db2set DB2_LOGGER_NON_BUFFERED_IO=OFF

# On UNIX platforms, the administration notification log is a text file called instance.nfy.
# level 1 Only fatal and unrecoverable errors are logged.
db2 update dbm cfg using notifylevel 1
db2 connect to "${db}"

# Maximum number of concurrent applications that can be connected (both local and remote) to a database
db2 update db cfg for "${db}" using maxappls 256
db2 update db cfg for "${db}" using avg_appls 256
db2 update db cfg for "${db}" using locklist 4000
db2 update db cfg for "${db}" using maxlocks 40

# specify a string of up to 242 bytes to change the location where the log files are stored.
# You can use the database system monitor to track the number of I/Os related to database logging.
# The monitor elements log_reads (number of log pages read) and log_writes (number of log pages written)
# return the amount of I/O activity related to database logging.
mkdir -p "${log_path}"
db2 update db cfg for "${db}" using newlogpath "${log_path}" || true

# size of each primary and secondary log file
# A log file that is too small can affect system performance because of the overhead of archiving
# old log files, allocating new log files, and waiting for a usable log file
db2 update db cfg for "${db}" using logfilsiz 10000 || true
# 6 logs * 5000 = 30K pages *4K= 120MB
db2 update db cfg for "${db}" using logprimary 10 || true
db2 update db cfg for "${db}" using logsecond 0
db2 update db cfg for "${db}" using num_ioservers 1 || true
db2 update db cfg for "${db}" using num_iocleaners 1 || true
db2 update db cfg for "${db}" using dft_queryopt 0
#db2 update db cfg for "${db}" using mincommit 2
#db2 update db cfg for "${db}" using logbufsz 132
db2 update db cfg for "${db}" using AUTO_MAINT OFF

db2 reorgchk update statistics

db2 reorg indexes all for table "accountejb"
db2 reorg indexes all for table "accountprofileejb"
db2 reorg indexes all for table "holdingejb"
db2 reorg indexes all for table "keygenejb"
db2 reorg indexes all for table "orderejb"
db2 reorg indexes all for table "quoteejb"

db2 connect reset
db2 terminate
