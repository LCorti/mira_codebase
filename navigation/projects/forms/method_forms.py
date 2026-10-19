from typing import ClassVar

from django import forms

from projects.models import Method, Test


class MethodForm(forms.ModelForm):
    """
    Form for creating and editing methods.
    Matches the structure expected by the processing service.
    """

    # Additional fields for easier UI input
    metrics_list = forms.CharField(
        required=False,
        widget=forms.TextInput(
            attrs={
                "class": "w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-colors duration-200",
                "placeholder": "Enter metrics (comma-separated)",
            }
        ),
        help_text="Comma-separated list of metrics",
    )

    dataset_split = forms.ChoiceField(
        choices=[
            ("train", "Train"),
            ("validation", "Validation"),
            ("test", "Test"),
        ],
        initial="train",
        widget=forms.Select(
            attrs={
                "class": "w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-colors duration-200"
            }
        ),
        help_text="Dataset split to use",
    )

    input_column = forms.CharField(
        required=False,
        initial="input",
        widget=forms.TextInput(
            attrs={
                "class": "w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-colors duration-200",
                "placeholder": "input",
            }
        ),
        help_text="Input column name in dataset",
    )

    reference_column = forms.CharField(
        required=False,
        initial="label",
        widget=forms.TextInput(
            attrs={
                "class": "w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-colors duration-200",
                "placeholder": "label",
            }
        ),
        help_text="Label column name in dataset",
    )

    class Meta:
        model = Method
        fields: ClassVar[list[str]] = ["name", "description", "test"]
        widgets: ClassVar[dict[str, forms.Widget]] = {
            "name": forms.TextInput(
                attrs={
                    "class": "w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-colors duration-200",
                    "placeholder": "Enter method name",
                }
            ),
            "description": forms.Textarea(
                attrs={
                    "class": "w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-colors duration-200",
                    "rows": 4,
                    "placeholder": "Enter method description",
                }
            ),
            "test": forms.Select(
                attrs={
                    "class": "w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-colors duration-200"
                }
            ),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only show tests that can be selected
        self.fields["test"].queryset = Test.objects.all().order_by("name")

        if self.instance and self.instance.pk:
            # Pre-populate form fields from instance data
            if self.instance.metrics:
                self.fields["metrics_list"].initial = ", ".join(self.instance.metrics)

            params = self.instance.parameters or {}
            if "dataset" in params:
                dataset_config = params["dataset"]
                self.fields["dataset_split"].initial = dataset_config.get(
                    "split", "train"
                )
                self.fields["input_column"].initial = dataset_config.get(
                    "input_column", "input"
                )
                self.fields["reference_column"].initial = dataset_config.get(
                    "reference_column", "label"
                )

    def clean_name(self):
        """Validate method name."""
        name = self.cleaned_data["name"]
        if len(name) < 3:
            raise forms.ValidationError(
                "Method name must be at least 3 characters long."
            )
        return name

    def save(self, commit=True):
        """Save the method with parameters from form fields."""
        method = super().save(commit=False)

        # Update metrics list
        if self.cleaned_data.get("metrics_list"):
            metrics_list = [
                metric.strip()
                for metric in self.cleaned_data["metrics_list"].split(",")
                if metric.strip()
            ]
            method.metrics = metrics_list
        else:
            method.metrics = []

        # Update parameters with form data
        params = method.parameters.copy() if method.parameters else {}

        # Update dataset configuration
        params["dataset"] = {
            "split": self.cleaned_data.get("dataset_split", "train"),
            "input_column": self.cleaned_data.get("input_column", "input"),
            "reference_column": self.cleaned_data.get("reference_column", "label"),
        }

        # Initialize metrics configuration if metrics are provided
        if method.metrics:
            params["metrics"] = {metric: {} for metric in method.metrics}
        else:
            params["metrics"] = {}

        method.parameters = params
        method.diagnostic_only = method.test.name in [
            "LXT",
            "CC-SHAP",
            "SentEval",
            "holistic_bias",
        ]

        if commit:
            method.save()
        return method


class MethodEditForm(forms.ModelForm):
    """
    Form for editing existing methods (excludes test field since it's already set).
    Only includes fields that are actually saved to the database.
    All configuration data comes from the JSON parameters field.
    """

    # JSON parameters field (this contains the complete configuration)
    parameters = forms.CharField(
        required=False,
        widget=forms.HiddenInput(),
        help_text="Complete method parameters as JSON",
    )

    class Meta:
        model = Method
        fields: ClassVar[list[str]] = [
            "name",
            "description",
            "parameters",
        ]  # Only include actual database fields
        widgets: ClassVar[dict[str, forms.Widget]] = {
            "name": forms.TextInput(
                attrs={
                    "class": "w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-colors duration-200",
                    "placeholder": "Enter method name",
                }
            ),
            "description": forms.Textarea(
                attrs={
                    "class": "w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-colors duration-200",
                    "rows": 4,
                    "placeholder": "Enter method description",
                }
            ),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.instance and self.instance.pk and self.instance.parameters:
            # Pre-populate JSON parameters field
            import json

            self.fields["parameters"].initial = json.dumps(self.instance.parameters)

    def clean_name(self):
        """Validate method name."""
        name = self.cleaned_data["name"]
        if len(name) < 3:
            raise forms.ValidationError(
                "Method name must be at least 3 characters long."
            )
        return name

    def save(self, commit=True):
        """Save the method with parameters from JSON field."""
        import json

        method = super().save(commit=False)

        # Use JSON parameters if provided
        parameters_json = self.cleaned_data.get("parameters")

        if parameters_json:
            try:
                # Parse the JSON parameters
                params = json.loads(parameters_json)
                method.parameters = params

                # Update metrics list from parameters if available
                if "metrics" in params and isinstance(params["metrics"], dict):
                    new_metrics = list(params["metrics"].keys())
                    # Always update metrics from the JSON parameters
                    # An empty dict {} means the user intentionally removed all metrics
                    method.metrics = new_metrics
                else:
                    # No metrics in parameters, clear metrics list
                    method.metrics = []

            except json.JSONDecodeError as e:
                # If JSON parsing fails, log error but continue
                print(f"Warning: Could not parse parameters JSON: {e}")
                # Keep existing parameters if JSON is invalid
        else:
            # No parameters provided - this shouldn't normally happen
            print("Warning: No parameters provided in form submission")

        if commit:
            method.save()
        return method


class MethodSyncForm(forms.Form):
    """
    Form for syncing methods with processing service.
    """

    force_sync = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
        help_text="Force sync even if already synced",
    )
