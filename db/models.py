from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chat_sessions: Mapped[list['ChatSession']] = relationship('ChatSession', back_populates='user')
    plant_series: Mapped[list['PlantSeries']] = relationship('PlantSeries', back_populates='user')


class ChatSession(Base):
    __tablename__ = 'chat_sessions'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'), index=True)
    title: Mapped[str] = mapped_column(String(255), default='Новый чат')
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user: Mapped[User] = relationship('User', back_populates='chat_sessions')
    messages: Mapped[list['ChatMessage']] = relationship(
        'ChatMessage',
        back_populates='session',
        cascade='all, delete-orphan',
        order_by='ChatMessage.id',
    )


class ChatMessage(Base):
    __tablename__ = 'chat_messages'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[int] = mapped_column(ForeignKey('chat_sessions.id'), index=True)
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    run_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped[ChatSession] = relationship('ChatSession', back_populates='messages')


class PlantSeries(Base):
    __tablename__ = 'plant_series'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey('users.id'), nullable=True, index=True)
    tenant_id: Mapped[str] = mapped_column(String(128), default='default', index=True)
    crop: Mapped[str] = mapped_column(String(64), default='Unknown')
    name: Mapped[str] = mapped_column(String(255), default='Growth Series')
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped[User | None] = relationship('User', back_populates='plant_series')
    observations: Mapped[list['PlantObservation']] = relationship(
        'PlantObservation',
        back_populates='series',
        cascade='all, delete-orphan',
        order_by='PlantObservation.frame_index',
    )


class PlantObservation(Base):
    __tablename__ = 'plant_observations'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    series_id: Mapped[int] = mapped_column(ForeignKey('plant_series.id'), index=True)
    frame_index: Mapped[int] = mapped_column(Integer)
    run_id: Mapped[str] = mapped_column(String(255), index=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    phi_score: Mapped[float] = mapped_column(Float, default=0.0)
    phi_status: Mapped[str] = mapped_column(String(32), default='Risk')
    metrics_json: Mapped[str] = mapped_column(Text, default='{}')

    series: Mapped[PlantSeries] = relationship('PlantSeries', back_populates='observations')
