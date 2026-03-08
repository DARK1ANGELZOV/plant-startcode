from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import ChatMessage, ChatSession
from utils.schemas import ChatMessageResponse, ChatSearchHitResponse, ChatSessionResponse


class ChatService:
    def create_session(self, db: Session, user_id: int, title: str = 'Новый чат') -> ChatSession:
        session = ChatSession(user_id=user_id, title=title)
        db.add(session)
        db.commit()
        db.refresh(session)
        return session

    def list_sessions(self, db: Session, user_id: int, limit: int = 100) -> list[ChatSessionResponse]:
        rows = db.scalars(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
            .limit(limit)
        ).all()
        return [
            ChatSessionResponse(
                id=row.id,
                title=self._normalize_ru_text(row.title),
                created_at=self._as_iso(row.created_at),
                updated_at=self._as_iso(row.updated_at),
            )
            for row in rows
        ]

    def delete_session(self, db: Session, user_id: int, session_id: int) -> bool:
        session = self.get_session(db, user_id=user_id, session_id=session_id)
        if session is None:
            return False
        db.delete(session)
        db.commit()
        return True

    def get_session(self, db: Session, user_id: int | None, session_id: int, allow_any: bool = False) -> ChatSession | None:
        query = select(ChatSession).where(ChatSession.id == session_id)
        if not allow_any:
            query = query.where(ChatSession.user_id == user_id)
        return db.scalar(query)

    def append_message(
        self,
        db: Session,
        user_id: int,
        session_id: int,
        role: str,
        content: str,
        run_id: str | None = None,
    ) -> ChatMessage:
        session = self.get_session(db, user_id, session_id)
        if session is None:
            raise ValueError('Chat session not found.')

        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            run_id=run_id,
        )
        db.add(message)

        session.updated_at = datetime.now(timezone.utc)
        db.add(session)

        db.commit()
        db.refresh(message)
        return message

    def get_messages(
        self,
        db: Session,
        user_id: int | None,
        session_id: int,
        limit: int = 200,
        allow_any: bool = False,
    ) -> list[ChatMessageResponse]:
        session = self.get_session(db, user_id, session_id, allow_any=allow_any)
        if session is None:
            raise ValueError('Chat session not found.')

        rows = db.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id.asc())
            .limit(limit)
        ).all()
        return [
            ChatMessageResponse(
                id=row.id,
                role=row.role,
                content=self._normalize_ru_text(row.content),
                run_id=row.run_id,
                created_at=self._as_iso(row.created_at),
            )
            for row in rows
        ]

    def search_messages(
        self,
        db: Session,
        user_id: int,
        query: str,
        limit: int = 50,
    ) -> list[ChatSearchHitResponse]:
        q = (query or '').strip()
        if not q:
            return []

        rows = db.execute(
            select(
                ChatSession.id,
                ChatSession.title,
                ChatMessage.id,
                ChatMessage.content,
                ChatMessage.created_at,
            )
            .join(ChatMessage, ChatMessage.session_id == ChatSession.id)
            .where(ChatSession.user_id == user_id)
            .where(ChatMessage.content.ilike(f'%{q}%'))
            .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
            .limit(limit)
        ).all()

        result: list[ChatSearchHitResponse] = []
        for session_id, title, message_id, content, created_at in rows:
            result.append(
                ChatSearchHitResponse(
                    session_id=int(session_id),
                    title=self._normalize_ru_text(str(title or 'Новый чат')),
                    message_id=int(message_id),
                    excerpt=self._make_excerpt(self._normalize_ru_text(str(content or '')), q),
                    created_at=self._as_iso(created_at),
                )
            )
        return result

    @staticmethod
    def _make_excerpt(content: str, query: str, radius: int = 70) -> str:
        text = ' '.join(content.split())
        if not text:
            return ''
        q = query.lower()
        pos = text.lower().find(q)
        if pos < 0:
            return text[:radius * 2].strip()
        start = max(0, pos - radius)
        end = min(len(text), pos + len(query) + radius)
        excerpt = text[start:end].strip()
        if start > 0:
            excerpt = '... ' + excerpt
        if end < len(text):
            excerpt = excerpt + ' ...'
        return excerpt

    @staticmethod
    def _as_iso(value) -> str:
        if hasattr(value, 'isoformat'):
            return value.isoformat()
        return str(value)

    @staticmethod
    def _normalize_ru_text(value: str) -> str:
        text = str(value or '')
        if not text:
            return text

        # Common mojibake symbols when UTF-8 text was decoded as cp1251.
        suspect = {
            *range(0x0402, 0x0410),
            *range(0x0452, 0x0460),
        }
        if not any(ord(ch) in suspect for ch in text):
            return text

        try:
            fixed = text.encode('cp1251').decode('utf-8')
        except Exception:
            return text

        def quality(s: str) -> tuple[int, int]:
            bad = sum(1 for ch in s if ord(ch) in suspect)
            low = s.lower()
            good = sum(low.count(bg) for bg in ('ст', 'но', 'ни', 'на', 'то', 'пр', 'по', 'ро', 'ен'))
            return bad, good

        bad_src, good_src = quality(text)
        bad_fix, good_fix = quality(fixed)
        if bad_fix < bad_src and good_fix >= good_src:
            return fixed
        return text
