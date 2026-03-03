from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import User


pwd_context = CryptContext(schemes=['pbkdf2_sha256'], deprecated='auto')
JWT_SECRET = os.getenv('JWT_SECRET', 'change-me-in-production')
JWT_ALGO = 'HS256'
JWT_EXPIRE_MINUTES = int(os.getenv('JWT_EXPIRE_MINUTES', '1440'))


class AuthService:
    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        return pwd_context.verify(password, password_hash)

    @staticmethod
    def create_access_token(user_id: int, email: str) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            'sub': str(user_id),
            'email': email,
            'iat': int(now.timestamp()),
            'exp': int((now + timedelta(minutes=JWT_EXPIRE_MINUTES)).timestamp()),
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

    @staticmethod
    def decode_token(token: str) -> dict:
        try:
            return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        except JWTError as exc:
            raise ValueError('Invalid token.') from exc

    def register(self, db: Session, email: str, password: str) -> User:
        existing = db.scalar(select(User).where(User.email == email))
        if existing is not None:
            raise ValueError('Email already registered.')

        user = User(email=email, password_hash=self.hash_password(password))
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    def login(self, db: Session, email: str, password: str) -> User:
        user = db.scalar(select(User).where(User.email == email))
        if user is None or not self.verify_password(password, user.password_hash):
            raise ValueError('Invalid email or password.')
        return user

    def change_password(self, db: Session, user_id: int, old_password: str, new_password: str) -> None:
        user = self.get_user_by_id(db, user_id)
        if user is None:
            raise ValueError('User not found.')
        if not self.verify_password(old_password, user.password_hash):
            raise ValueError('Old password is incorrect.')
        user.password_hash = self.hash_password(new_password)
        db.add(user)
        db.commit()

    def get_user_by_id(self, db: Session, user_id: int) -> User | None:
        return db.get(User, user_id)
