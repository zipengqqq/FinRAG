from sqlalchemy import BigInteger, Column, Integer, String, DateTime, Numeric
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()

class FileModel(Base):
    __tablename__ = 'file'

    id = Column(BigInteger, primary_key=True, autoincrement=False, comment='文件ID')
    minio_url = Column(String(200), nullable=True, comment='文件minio地址')
    status = Column(Integer, nullable=True, comment='文件状态；0文件待解析，1文件正在解析，2文件解析成功，3文件解析失败')
    file_name = Column(String(50), nullable=True, comment='文件名')
    size = Column(Numeric(precision=10, scale=2), nullable=True, comment='文件大小，单位是MB')
    type = Column(Integer, nullable=True, comment='文件类型，0表PDF')
    create_time = Column(DateTime, nullable=True, comment='创建时间')
    delete_flag = Column(Integer, nullable=True, comment='删除标识，0未删除，1已删除')

    def __repr__(self):
        columns = [col.name for col in self.__table__.columns]
        attrs = ', '.join(f"{col}={repr(getattr(self, col))}" for col in columns)
        return f"<{self.__class__.__name__}({attrs})>"

    def to_dict(self):
        columns = [col.name for col in self.__table__.columns]
        data = {col: getattr(self, col) for col in columns}
        if data.get("id") is not None:
            data["id"] = str(data["id"])
        return data
