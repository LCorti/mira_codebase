// MongoDB initialization script
// This script runs automatically when the container starts for the first time
// It creates the application user with the necessary permissions

db = db.getSiblingDB('admin');

// Get credentials from environment variables
const username = process.env.MONGO_INITDB_ROOT_USERNAME;
const password = process.env.MONGO_INITDB_ROOT_PASSWORD;

print('========================================');
print('Creating MongoDB user for MIRA application...');
print('Username: ' + username);
print('========================================');

// Create user with readWrite and dbAdmin roles on mira-processing database
try {
  db.createUser({
    user: username,
    pwd: password,
    roles: [
      {
        role: 'readWrite',
        db: 'mira-processing'
      },
      {
        role: 'dbAdmin',
        db: 'mira-processing'
      }
    ]
  });
  print('> User created successfully');
} catch (error) {
  if (error.code === 51003) {
    print('- User already exists, skipping creation...');
  } else {
    print('!! Error creating user: ' + error);
    throw error;
  }
}

print('========================================');
print('MongoDB initialization complete!');
print('========================================');
