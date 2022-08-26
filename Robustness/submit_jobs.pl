#! /usr/bin/perl -w

# split the fits files in a directory into several gobs of size $MAXIMAGES
use lib "$ENV{HESSROOT}/summary/scripts";
use HESS;
use strict;
use warnings;
use English;
$HESS::ECAP_QUEUE = 1;



my $myCmd = "module unload python-packages;  python /home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/nuisance_summary/Robustness/submission_overview.py  ";   
my $qid = &HESS::submit_to_queue_when_possible( $myCmd  );

