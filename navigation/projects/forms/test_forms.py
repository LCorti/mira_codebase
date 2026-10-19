import json
from typing import ClassVar

from django import forms

from projects.models import Test


class TestForm(forms.ModelForm):
    """
    Form for creating and editing tests.
    Matches the structure expected by the processing service.
    """

    name = forms.CharField(
        required=True,
        max_length=200,
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "Enter test name"}
        ),
        help_text="Name of the test",
    )

    description = forms.CharField(
        required=False,
        widget=forms.Textarea(
            attrs={
                "class": "form-control",
                "rows": 4,
                "placeholder": "Enter test description",
            }
        ),
        help_text="Description of the test",
    )

    project_id = forms.CharField(required=True, widget=forms.HiddenInput())

    metrics = forms.CharField(
        required=False,
        widget=forms.HiddenInput(),
        help_text="JSON string containing metrics per dataset",
    )

    class Meta:
        model = Test
        fields: ClassVar[list[str]] = ["name", "description", "project_id", "metrics"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.instance and self.instance.pk:
            self.fields["name"].initial = self.instance.name
            self.fields["description"].initial = self.instance.description
            self.fields["project_id"].initial = self.instance.project_id

    def clean_name(self):
        """Validate test name."""
        name = self.cleaned_data["name"]
        if len(name) < 3:
            raise forms.ValidationError("Test name must be at least 3 characters long.")
        return name

    def clean_metrics(self):
        """Parse and validate metrics JSON."""
        metrics_str = self.cleaned_data.get("metrics", "")
        if not metrics_str.strip():
            return {}

        try:
            metrics = json.loads(metrics_str)
            # Validate that metrics is a dictionary with dataset names as keys
            if not isinstance(metrics, dict):
                raise forms.ValidationError("Metrics must be a JSON object.")

            # Validate that each value is a list
            for dataset_name, metric_list in metrics.items():
                if not isinstance(metric_list, list):
                    raise forms.ValidationError(
                        f"Metrics for dataset '{dataset_name}' must be a list."
                    )

            return metrics
        except json.JSONDecodeError as e:
            raise forms.ValidationError(f"Invalid JSON format for metrics: {e}")

    def clean_datasets(self):
        """Extract datasets from metrics keys."""
        # Get the metrics data (which should be cleaned first)
        metrics = self.cleaned_data.get("metrics", {})

        if isinstance(metrics, dict):
            return list(metrics.keys())

        return []

    def clean(self):
        """Custom clean method to set datasets based on metrics."""
        cleaned_data = super().clean()

        # Extract datasets from metrics
        metrics = cleaned_data.get("metrics", {})
        if metrics:
            cleaned_data["datasets"] = list(metrics.keys())
        else:
            cleaned_data["datasets"] = []

        return cleaned_data

    def save(self, commit=True):
        self.instance.name = self.cleaned_data.get("name", "")
        self.instance.description = self.cleaned_data.get("description", "")

        self.instance.datasets = self.cleaned_data.get("datasets", [])
        self.instance.metrics = self.cleaned_data.get("metrics", {})

        # Set project ID
        self.instance.project_id = self.cleaned_data.get("project_id", None)

        self.instance.parameters = self.instance.get_default_parameters()

        print(f"Instance name: {self.instance.name}")
        self.instance.diagnostic_only = self.instance.name in [
            "LXT",
            "CC-SHAP",
            "SentEval",
            "holistic_bias",
        ]
        print(f"Diagnostic only: {self.instance.diagnostic_only}")

        return super().save(commit=commit)


class TestSyncForm(forms.Form):
    """
    Form for syncing tests with processing service.
    """

    force_sync = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
        help_text="Force sync even if already synced",
    )
