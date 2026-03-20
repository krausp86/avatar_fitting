import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async


class FittingConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.job_id   = self.scope['url_route']['kwargs']['job_id']
        self.group_name = f'job_{self.job_id}'
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def fitting_progress(self, event):
        """Receive progress from fitting pipeline and forward to browser."""
        await self.send(text_data=json.dumps(event['data']))

    async def fitting_complete(self, event):
        await self.send(text_data=json.dumps({'type': 'complete', **event['data']}))

    async def fitting_error(self, event):
        await self.send(text_data=json.dumps({'type': 'error', **event['data']}))
