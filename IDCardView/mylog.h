#ifndef _MYLOG_H
#define _MYLOG_H

#include<android/log.h>
#define LOG    "IDCard"
#define LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG,__VA_ARGS__)
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG,__VA_ARGS__)


#endif
