from django.urls import path

from projects.views import project, test

urlpatterns = [
    path('tests/', test.create, name='create_test'),  # /projects/tests/
    
    # Open project route
    path('open/', project.open, name='projects_open'),  # /projects/open
    path('create/', project.create, name='projects_create'),  # /projects/create/
    
    # Pull all data from processing service
    path('pull-all/', project.pull_all_from_processing_service, name='projects_pull_all'),  # /projects/pull-all/
    
    # Project report routes (must be before general section routes)
    path('<str:project_id>/structure/', project.get_project_structure, name='projects_get_structure'),  # /projects/<id>/structure/
    path('<str:project_id>/generate-project-report/', project.generate_project_report, name='projects_generate_project_report'),  # /projects/<id>/generate-project-report/
    
    # Project management routes
    path('<str:project_id>/edit/', project.edit, name='projects_edit'),  # /projects/<id>/edit/
    path('<str:project_id>/delete/', project.delete, name='projects_delete'),  # /projects/<id>/delete/
    
    path('<str:project_id>/', project.show, name='projects_show'),  # /projects/<id>/
    path('<str:project_id>/<str:section>/', project.show, name='projects_show_section'),  # /projects/<id>/<section>/
    path('<str:project_id>/<str:section>/test/<str:test_id>/', project.show, name='projects_show_test'),  # /projects/<id>/<section>/test/<test_id>/
    path('<str:project_id>/<str:section>/method/<str:method_id>/', project.show, name='projects_show_method'),  # /projects/<id>/<section>/method/<method_id>/
]
