from django.urls import path

from projects.views import test

urlpatterns = [
    path('<str:project_id>/tests/<str:test_id>/update-parameters/', test.update_parameters, name='test_update_parameters'),  # /projects/<project_id>/tests/<test_id>/update-parameters/

    path('<str:project_id>/tests/create/', test.create, name='create_test'),  # /projects/<project_id>/tests/create/
    path('<str:project_id>/tests/<str:test_id>/edit/', test.edit, name='edit_test'),  # /projects/<project_id>/tests/<test_id>/edit/
]
    