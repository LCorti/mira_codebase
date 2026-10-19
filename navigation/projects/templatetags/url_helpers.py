from django import template
from django.urls import reverse

register = template.Library()

@register.simple_tag
def project_test_url(project_id, section='search', test_id=None):
    """
    Generate URL for a project with optional test focus
    """
    if test_id:
        return reverse('projects_show_test', kwargs={
            'project_id': project_id,
            'section': section,
            'test_id': test_id
        })
    elif section:
        return reverse('projects_show_section', kwargs={
            'project_id': project_id,
            'section': section
        })
    else:
        return reverse('projects_show', kwargs={'project_id': project_id})

@register.simple_tag
def project_method_url(project_id, section='search', method_id=None):
    """
    Generate URL for a project with optional method focus
    """
    if method_id:
        return reverse('projects_show_method', kwargs={
            'project_id': project_id,
            'section': section,
            'method_id': method_id
        })
    elif section:
        return reverse('projects_show_section', kwargs={
            'project_id': project_id,
            'section': section
        })
    else:
        return reverse('projects_show', kwargs={'project_id': project_id})
