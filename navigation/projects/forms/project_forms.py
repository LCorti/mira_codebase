import json
from typing import ClassVar

from django import forms

from projects.models import Project


class ProjectForm(forms.ModelForm):
    """
    Form for creating and editing projects.
    Handles all fields from the setup wizard including model selection,
    datasets, tests, and parameters.
    """

    # Wizard step 1: Model selection
    selected_model = forms.CharField(
        required=False,
        widget=forms.HiddenInput(),
        help_text="Selected model from the wizard",
    )

    custom_model = forms.CharField(
        required=False,
        widget=forms.HiddenInput(),
        help_text="Custom model identifier from HuggingFace",
    )

    # Wizard step 2: Dataset and metric selection
    datasets = forms.CharField(
        required=False,
        widget=forms.HiddenInput(),
        help_text="Comma-separated list of selected datasets",
    )

    metrics = forms.CharField(
        required=False,
        widget=forms.HiddenInput(),
        help_text="JSON string containing metrics per dataset",
    )

    # Wizard step 3: Test selection
    test_ids = forms.CharField(
        required=False,
        widget=forms.HiddenInput(),
        help_text="Comma-separated list of selected tests",
    )

    # Hidden field to hold the model_name (populated from selected_model or custom_model)
    model_name = forms.CharField(
        required=False,
        widget=forms.HiddenInput(),
        help_text="Model name derived from selected_model or custom_model",
    )

    # Wizard step 4: Project info (mapped to model fields)
    name = forms.CharField(
        required=True,
        max_length=200,
        widget=forms.TextInput(
            attrs={
                "class": "w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-colors duration-200",
                "placeholder": "Enter a descriptive name for your project",
            }
        ),
        help_text="This will be used to identify your project in the dashboard",
    )

    description = forms.CharField(
        required=False,
        widget=forms.Textarea(
            attrs={
                "class": "w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-colors duration-200",
                "rows": 3,
                "placeholder": "Optional description of your project's purpose and goals",
            }
        ),
    )

    # Wizard step 5: Parameters
    parameters = forms.CharField(
        required=False,
        widget=forms.Textarea(
            attrs={
                "class": "w-full px-4 py-3 font-mono text-sm bg-gray-50 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-colors duration-200",
                "rows": 15,
                "placeholder": "Parameters will be auto-populated based on your selections...",
            }
        ),
        help_text="JSON parameters for the project",
    )

    class Meta:
        model = Project
        fields: ClassVar[list[str]] = [
            "name",
            "model_name",
            "description",
            "auto_run",
            "parameters",
            "test_ids",
        ]
        widgets: ClassVar[dict[str, forms.Widget]] = {
            "auto_run": forms.CheckboxInput(
                attrs={
                    "class": "w-4 h-4 text-primary bg-gray-100 border-gray-300 rounded focus:ring-primary focus:ring-2",
                    "value": "true",
                }
            ),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.instance and self.instance.pk:
            # Populate wizard fields from existing instance
            params = self.instance.parameters or {}
            self.fields["name"].initial = self.instance.name
            self.fields["description"].initial = self.instance.description
            self.fields["parameters"].initial = json.dumps(params, indent=2)
            self.fields["model_name"].initial = self.instance.model_name

            # Extract datasets from parameters
            if "dataset" in params:
                datasets = list(params["dataset"].keys())
                self.fields["datasets"].initial = ",".join(datasets)

            # Set test IDs
            if self.instance.test_ids:
                self.fields["test_ids"].initial = ",".join(self.instance.test_ids)

    def clean(self):
        """Custom clean method to populate model_name from wizard fields."""
        cleaned_data = super().clean()

        # Get the submitted values
        selected_model = cleaned_data.get("selected_model", "")
        custom_model = cleaned_data.get("custom_model", "")

        # Prioritize custom_model if provided, otherwise use selected_model
        model_name = (
            custom_model.strip() if custom_model.strip() else selected_model.strip()
        )

        if not model_name:
            raise forms.ValidationError(
                "Please select a model or provide a custom model name."
            )

        # Set model_name in cleaned_data
        cleaned_data["model_name"] = model_name

        return cleaned_data

    def clean_name(self):
        """Validate project name."""
        name = self.cleaned_data.get("name", "")
        if len(name) < 3:
            raise forms.ValidationError(
                "Project name must be at least 3 characters long."
            )
        return name

    def clean_parameters(self):
        """Validate and parse JSON parameters."""
        parameters_str = self.cleaned_data.get("parameters", "")
        if not parameters_str.strip():
            # Return default parameters if empty
            from projects.models.project import get_default_project_parameters

            return get_default_project_parameters()

        try:
            parameters = json.loads(parameters_str)
            return parameters
        except json.JSONDecodeError as e:
            raise forms.ValidationError(f"Invalid JSON format: {e}")

    def clean_datasets(self):
        """Parse selected datasets."""
        # Handle both comma-separated string and list inputs
        datasets_data = (
            self.data.getlist("datasets")
            if hasattr(self.data, "getlist")
            else self.cleaned_data.get("datasets", "")
        )

        if isinstance(datasets_data, list):
            # If we get a list (from getlist), filter out empty strings
            datasets = [d.strip() for d in datasets_data if d.strip()]
            return datasets
        else:
            # Fallback to comma-separated string parsing
            datasets_str = str(datasets_data)
            if not datasets_str.strip():
                return []
            return [d.strip() for d in datasets_str.split(",") if d.strip()]

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

    def clean_test_ids(self):
        """Parse selected tests."""
        # Handle both comma-separated string and list inputs
        tests_data = (
            self.data.getlist("test_ids")
            if hasattr(self.data, "getlist")
            else self.cleaned_data.get("test_ids", "")
        )

        if isinstance(tests_data, list):
            # If we get a list (from getlist), filter out empty strings
            tests = [t.strip() for t in tests_data if t.strip()]
            return tests
        else:
            # Fallback to comma-separated string parsing
            tests_str = str(tests_data)
            if not tests_str.strip():
                return []
            return [t.strip() for t in tests_str.split(",") if t.strip()]

    def save(self, commit=True):
        """Save the project with all wizard data properly mapped."""
        # Map wizard fields to model fields
        self.instance.name = self.cleaned_data.get("name", "")
        self.instance.description = self.cleaned_data.get("description", "")

        # Set datasets and metrics directly on the model instance
        self.instance.datasets = self.cleaned_data.get("datasets", [])
        self.instance.metrics = self.cleaned_data.get("metrics", {})

        # Set the complete parameters
        self.instance.parameters = self.cleaned_data.get("parameters", {})

        # Set test IDs
        self.instance.test_ids = self.cleaned_data.get("test_ids", [])

        return super().save(commit=commit)


class ProjectSyncForm(forms.Form):
    """
    Form for syncing projects with processing service.
    """

    force_sync = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
        help_text="Force sync even if already synced",
    )
