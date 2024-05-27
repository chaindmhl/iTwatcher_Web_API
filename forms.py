from django import forms
from .models import PlateLog, DownloadRequest
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class PlateLogForm(forms.ModelForm):
    class Meta:
        model = PlateLog
        fields = '__all__'  # Include all fields from the model

    # You can customize form fields or widgets here if needed

class SignUpForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=True, label='First Name', widget=forms.TextInput(attrs={'placeholder': 'Required'}))
    last_name = forms.CharField(max_length=30, required=True, label='Last Name', widget=forms.TextInput(attrs={'placeholder': 'Required'}))
    email = forms.EmailField(max_length=254, required=True, label='Email', widget=forms.EmailInput(attrs={'placeholder': 'Enter a valid email address.'}))
    username = forms.CharField(max_length=20, required=True, label='Username', widget=forms.TextInput(attrs={'placeholder': 'Required'}))
    password1 = forms.CharField(label='Password', widget=forms.PasswordInput(attrs={'placeholder': 'Password must at least 8 characters.'}))
    password2 = forms.CharField(label='Password confirmation', widget=forms.PasswordInput(attrs={'placeholder': 'Enter password again.'}))

    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'email', 'username', 'password1', 'password2')

class DownloadRequestForm(forms.ModelForm):
    class Meta:
        model = DownloadRequest
        fields = '__all__'  # Include all fields from the model