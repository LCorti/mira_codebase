project_urls = __import__('projects.routes.projects', fromlist=['urlpatterns']).urlpatterns

test_urls = __import__('projects.routes.tests', fromlist=['urlpatterns']).urlpatterns

method_urls = __import__('projects.routes.methods', fromlist=['urlpatterns']).urlpatterns

urlpatterns = project_urls + test_urls + method_urls
