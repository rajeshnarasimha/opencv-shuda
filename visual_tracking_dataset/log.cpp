/*
 * Developer : Developer Jack Z.G. Tan(jack@statsmaster.com.hk)
 * Date : 01/11/2012
 * All code (c)2011 Statsmaster Ltd all rights reserved
 */

#include "log.h"
#include <stdarg.h>
#include <stdlib.h>

FILE *g_pLogFile = NULL;         // log file: lprintfs go here
static char gPrintBuf[8192];			// buffer for routines in this file
// so print routines don't use sgBuf

/* like printf but prints to the log file as well, it is is open. */
void lprintf (const char *args, ...)   // args like printf
{
	va_list arg;
	va_start(arg, args);
	vsprintf(gPrintBuf, args, arg);
	va_end(arg);
	printf("%s", gPrintBuf);
	fflush(stdout);     // flush so if there is a crash we can see what happened
	if (g_pLogFile)
	{
		fputs(gPrintBuf, g_pLogFile);
		fflush(g_pLogFile);
	}
}

/* 
 Like printf but prints to the log file only (and not to screen).
 Used for detailed stuff that we may want to know but we don't want to
 usually bother the user with.
*/

void logprintf (const char *args, ...) // args like printf
{
	if (g_pLogFile)
	{
		va_list arg;
		va_start(arg, args);
		vsprintf(gPrintBuf, args, arg);
		va_end(arg);
		fputs(gPrintBuf, g_pLogFile);
		fflush(g_pLogFile);
	}
}

