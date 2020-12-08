#!/usr/bin/expect
set host [lindex $argv 0]
set pass_wd [lindex $argv 1]
set cmd [lindex $argv 2]
spawn ssh $host
expect "password:"
send "$pass_wd\n"
expect "*$"
send "$cmd\n"
expect "*$"
interact
