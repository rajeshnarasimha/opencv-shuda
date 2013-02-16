/*
 * Developer : Developer Jack Z.G. Tan(jack@statsmaster.com.hk)
 * Date : 01/11/2012
 * All code (c)2011 Statsmaster Ltd all rights reserved
 */
#ifndef _LOG_H
#define _LOG_H

#include <stdio.h>

extern FILE *g_pLogFile;     // lprintfs go to this file

void lprintf (const char *args, ...) /* args like printf */;
void logprintf (const char *args, ...) /* args like printf */;

#endif