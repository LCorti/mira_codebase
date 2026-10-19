"""
Context processors for making data available globally in templates
"""

from django.conf import settings


def meta_info(request):
    """
    Add meta information to the context for use in templates
    """
    return {
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION,
        "build_date": settings.BUILD_DATE,
        "license": settings.LICENSE,
        "hf_home": settings.HF_HOME,
        "logging_level": settings.LOGGING_LEVEL,
        "django_debug": settings.DJANGO_DEBUG,
        "report_dir": settings.REPORT_DIR,
        "processing_service_ip": settings.PROCESSING_SERVICE_IP,
    }
